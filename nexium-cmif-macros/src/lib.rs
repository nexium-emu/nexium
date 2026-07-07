use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, ImplItem, ImplItemFn, ItemImpl, Lit, Meta, Pat, Type};

#[proc_macro_attribute]
pub fn command(_args: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn service(_args: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemImpl);
    let self_ty = input.self_ty.clone();

    let mut arms: Vec<TokenStream2> = Vec::new();

    for item in input.items.iter_mut() {
        let ImplItem::Fn(method) = item else {
            continue;
        };

        let Some(cmd_id) = extract_command_id(method) else {
            continue;
        };

        let arm = match build_dispatch_arm(cmd_id, method) {
            Ok(tokens) => tokens,
            Err(msg) => {
                return syn::Error::new_spanned(&method.sig.ident, msg)
                    .to_compile_error()
                    .into();
            }
        };

        method.attrs.retain(|a| !attr_is_command(a));

        arms.push(arm);
    }

    let expanded = quote! {
        #input

        impl #self_ty {
            pub fn dispatch_cmif(
                &mut self,
                cmd_id: u32,
                ctx: &mut ::nexium_cmif::DispatchCtx<'_>,
            ) -> ::core::option::Option<::nexium_cmif::DispatchOutcome> {
                match cmd_id {
                    #(#arms)*
                    _ => ::core::option::Option::None,
                }
            }
        }
    };

    expanded.into()
}

fn extract_command_id(method: &ImplItemFn) -> Option<u32> {
    for attr in &method.attrs {
        if !attr_is_command(attr) {
            continue;
        }
        if let Meta::List(list) = &attr.meta {
            let lit: Lit = match list.parse_args() {
                Ok(l) => l,
                Err(_) => continue,
            };
            if let Lit::Int(int) = lit {
                if let Ok(v) = int.base10_parse::<u32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn attr_is_command(attr: &syn::Attribute) -> bool {
    attr.path()
        .segments
        .last()
        .map(|s| s.ident == "command")
        .unwrap_or(false)
}

fn build_dispatch_arm(cmd_id: u32, method: &ImplItemFn) -> Result<TokenStream2, String> {
    let name = &method.sig.ident;
    let mut input_offset: usize = 0;
    let mut call_args: Vec<TokenStream2> = Vec::new();
    let mut pre_stmts: Vec<TokenStream2> = Vec::new();

    for (i, arg) in method.sig.inputs.iter().enumerate() {
        match arg {
            FnArg::Receiver(_) => continue,
            FnArg::Typed(pat_type) => {
                let arg_ident = format_ident!("__arg{}", i);
                let ty = &*pat_type.ty;
                let classification = classify_type(ty);
                match classification {
                    ArgKind::Primitive { size } => {
                        let off = input_offset;
                        pre_stmts.push(quote! {
                            let #arg_ident: #ty = ::nexium_cmif::read_in_arg(ctx.input_data, #off);
                        });
                        input_offset += size;
                        call_args.push(quote! { #arg_ident });
                    }
                    ArgKind::RecvBuffer => {
                        pre_stmts.push(quote! {
                            let #arg_ident = match ::nexium_cmif::RecvBuffer::from_ctx(ctx) {
                                ::core::option::Option::Some(b) => b,
                                ::core::option::Option::None => return ::core::option::Option::Some(
                                    ::nexium_cmif::DispatchOutcome::err(0xCE01)
                                ),
                            };
                        });
                        call_args.push(quote! { #arg_ident });
                    }
                    ArgKind::SendBuffer => {
                        pre_stmts.push(quote! {
                            let #arg_ident = match ::nexium_cmif::SendBuffer::from_ctx(ctx) {
                                ::core::option::Option::Some(b) => b,
                                ::core::option::Option::None => return ::core::option::Option::Some(
                                    ::nexium_cmif::DispatchOutcome::err(0xCE01)
                                ),
                            };
                        });
                        call_args.push(quote! { #arg_ident });
                    }
                    ArgKind::Ctx => {
                        call_args.push(quote! { ctx });
                    }
                    ArgKind::Unknown(name) => {
                        let pat_name = match &*pat_type.pat {
                            Pat::Ident(p) => p.ident.to_string(),
                            _ => format!("arg{}", i),
                        };
                        return Err(format!(
                            "unsupported #[command] argument type for `{}`: {}",
                            pat_name, name
                        ));
                    }
                }
            }
        }
    }

    Ok(quote! {
        #cmd_id => {
            #(#pre_stmts)*
            let __res = self.#name(#(#call_args),*);
            ::core::option::Option::Some(::nexium_cmif::finish(__res))
        }
    })
}

enum ArgKind {
    Primitive { size: usize },
    RecvBuffer,
    SendBuffer,
    Ctx,
    Unknown(String),
}

fn classify_type(ty: &Type) -> ArgKind {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            match name.as_str() {
                "u8" | "i8" => return ArgKind::Primitive { size: 1 },
                "u16" | "i16" => return ArgKind::Primitive { size: 2 },
                "u32" | "i32" => return ArgKind::Primitive { size: 4 },
                "u64" | "i64" => return ArgKind::Primitive { size: 8 },
                "RecvBuffer" => return ArgKind::RecvBuffer,
                "SendBuffer" => return ArgKind::SendBuffer,
                "DispatchCtx" => return ArgKind::Ctx,
                other => return ArgKind::Unknown(other.to_string()),
            }
        }
    }
    if let Type::Reference(r) = ty {
        if let Type::Path(p) = &*r.elem {
            if let Some(seg) = p.path.segments.last() {
                let name = seg.ident.to_string();
                if name == "DispatchCtx" {
                    return ArgKind::Ctx;
                }
                return ArgKind::Unknown(format!("&{}", name));
            }
        }
    }
    ArgKind::Unknown(quote! { #ty }.to_string())
}
